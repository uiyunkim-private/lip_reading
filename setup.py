from setuptools import setup, find_packages

setup(name='lip_reading',

      version='0.0.1',

      url='',

      license='MIT',

      author='Uiyun Kim',

      author_email='uiyunkim@kakao.com',

      description='lip reading',

      packages=find_packages(exclude=['tests']),

      long_description=open('README.md').read(),

      zip_safe=False,

      setup_requires=[''],

      test_suite='')